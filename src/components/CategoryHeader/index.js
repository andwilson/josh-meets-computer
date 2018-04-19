import React from "react";
import Link from "gatsby-link";
import styled from "styled-components";
import Img from "gatsby-image";

//import avatar from "../../images/avatar.jpg";
import github from "../../images/github-2.svg";
import linkedin from "../../images/linkedin-2.svg";
import instagram from "../../images/instagram-2.svg";

const GridContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 2fr;
  grid-gap: 10px;
  border-bottom: 1px solid grey;
  padding-bottom: 15px;
`;

const Social = styled.div`
  display: flex;
  justify-self: center;
  img {
    height: 35px;
    margin: 3px;
  }
  @media (max-width: 600px) {
    flex-direction: column;
  }
`;

const Profile = styled.div`
  display: flex;
  flex-direction: column;
  justify-self: center;
  align-items: center;
`;

const Info = styled.div`
  grid-template-rows: 1 / -1;
  grid-template-columns: 2;
  > h1 {
    color: black;
    margin: 0;
  }
  > h3 {
    color: grey;
    margin: 0;
    font-style: italic;
    font-weight: normal;
    margin-top: 8px;
  }
  > p {
    font-size: 12px;
  }
`;

const Avatar = styled(Img)`
  width: 150px;
  border-radius: 50%;
  justify-self: center;
  margin-bottom: 0.5em;
  border: 1px solid grey;
  &:hover {
     box-shadow: 0px 0px 2px grey
  }
  @media (max-width: 600px) {
    width: 110px;
  }
`;

export default ({ title, description, data }) => (
  <GridContainer>
    <Profile>
      <Avatar
        sizes={data.avatar.sizes}
      />
      <Social>
        <a href="https://github.com/JoshZastrow">
          <img src={github} />
        </a>
        <a href="https://www.linkedin.com/in/joshua-zastrow-b8131540/">
          <img src={linkedin} />
        </a>
        <a href="https://www.instagram.com/josh.zastrow/?hl=en">
          <img src={instagram} />
        </a>
      </Social>
    </Profile>
    <Info>
      <h1>{title}</h1>
      <h3>{data.allMarkdownRemark.totalCount} posts</h3>
      <p>{description}</p>
    </Info>
  </GridContainer>
);

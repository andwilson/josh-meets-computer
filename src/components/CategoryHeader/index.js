import React from "react";
import Link from "gatsby-link";
import styled from "styled-components";
import Img from "gatsby-image";

import avatar from "../../images/avatar.jpg";
import github from "../../images/github-2.svg";
import linkedin from "../../images/linkedin-2.svg";
import instagram from "../../images/instagram-2.svg";

const GridContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 2fr;
  border-bottom: 1px solid grey;
  padding-bottom: 15px;
  p {
    font-size: 14px;
  }
  h1 {
    margin: 0;
  }
  @media (max-width: 600px) {
    p {
      font-size: 12px;
    }
  }
`;

const Social = styled.div`
  display: flex;
`;

const Avatar = styled.img`
  height: 150px;
  border-radius: 50%;
  justify-self: center;
  margin-right: 10px;
  grid-row: 1 / -1;
  @media (max-width: 400px) {
    height: 100px;
  }
`;

const Icon = styled.img`
  height: 20px;
`;

export default ({ title, description }) => (
  <GridContainer>
    <Avatar src={avatar} />
    <Social>
      <a href="https://www.instagram.com/josh.zastrow/?hl=en"><Icon src={github} /></a>
      <a href="https://www.instagram.com/josh.zastrow/?hl=en"><Icon src={linkedin} /></a>
      <a href="https://www.instagram.com/josh.zastrow/?hl=en"><Icon src={instagram} /></a>
    </Social>
    <div>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  </GridContainer>
);
